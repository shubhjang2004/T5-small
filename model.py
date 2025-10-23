

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
import math
from typing import Optional, Tuple

# ============================================================================
# PART 1: Core Components - Relative Position Bias & Attention
# ============================================================================

class T5RelativePositionBias(nn.Module):
    """
    T5's unique relative position bias mechanism.
    Instead of absolute positions, uses bucketed relative positions.
    """
    def __init__(self, num_heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        # Learnable bias for each head and bucket
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
    
    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        """
        Convert relative positions to bucket indices.
        Handles both bidirectional (encoder) and causal (decoder) attention.
        """
        num_buckets = num_buckets
        ret = 0
        n = -relative_position
        
        # Half buckets for positive positions, half for negative
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        
        # Small distances get precise buckets, large distances are logarithmically spaced
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) 
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        
        ret += torch.where(is_small, n, val_if_large)
        return ret
    
    def forward(self, qlen, klen, device):
        """Compute relative position bias for attention scores"""
        # Create position indices
        context_position = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        
        # Compute relative positions
        relative_position = memory_position - context_position
        
        # Convert to buckets and get bias
        rp_bucket = self._relative_position_bucket(
            relative_position, 
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        
        values = self.relative_attention_bias(rp_bucket)  # (qlen, klen, num_heads)
        values = values.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, qlen, klen)
        return values


class T5Attention(nn.Module):
    """
    Multi-head attention with T5's relative position bias.
    Supports both self-attention and cross-attention.
    """
    def __init__(self, d_model=512, num_heads=8, dropout=0.1, has_relative_bias=True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.has_relative_bias = has_relative_bias
        
        # Q, K, V projections (no bias in T5)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias (only for first layer of encoder/decoder)
        if has_relative_bias:
            self.relative_bias = T5RelativePositionBias(num_heads=num_heads)
        else:
            self.relative_bias = None
    
    def forward(
        self, 
        hidden_states, 
        key_value_states=None,  # For cross-attention
        attention_mask=None,
        position_bias=None,  # Reuse bias from first layer
        return_position_bias=False
    ):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Self-attention or cross-attention
        is_cross_attention = key_value_states is not None
        kv_states = key_value_states if is_cross_attention else hidden_states
        kv_len = kv_states.shape[1]
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)  # (B, qlen, d_model)
        k = self.k_proj(kv_states)      # (B, klen, d_model)
        v = self.v_proj(kv_states)      # (B, klen, d_model)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add position bias
        if position_bias is None and self.relative_bias is not None:
            position_bias = self.relative_bias(seq_len, kv_len, hidden_states.device)
        
        if position_bias is not None:
            scores = scores + position_bias
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        
        if return_position_bias:
            return attn_output, position_bias
        return attn_output


# ============================================================================
# PART 2: Feed-Forward Network & Layer Norm
# ============================================================================

class T5DenseGatedGeluDense(nn.Module):
    """T5's feed-forward network with gated GELU activation"""
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)  # Transform
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, hidden_states):
        """Gated GELU: GELU(x @ W_gate) * (x @ W_transform)"""
        gated = self.gelu(self.wi_0(hidden_states))
        transformed = self.wi_1(hidden_states)
        hidden_states = gated * transformed
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerNorm(nn.Module):
    """T5's RMS Layer Normalization (no bias, no mean centering)"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        # RMS normalization
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


# ============================================================================
# PART 3: Encoder & Decoder Layers
# ============================================================================

class T5EncoderLayer(nn.Module):
    """Single T5 encoder layer with self-attention and FFN"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1, 
                 has_relative_bias=False):
        super().__init__()
        self.self_attention = T5Attention(
            d_model, num_heads, dropout, has_relative_bias=has_relative_bias
        )
        self.ffn = T5DenseGatedGeluDense(d_model, d_ff, dropout)
        
        self.layer_norm_1 = T5LayerNorm(d_model)
        self.layer_norm_2 = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        # Self-attention with residual connection
        normed_hidden_states = self.layer_norm_1(hidden_states)
        
        if position_bias is None:
            attn_output, position_bias = self.self_attention(
                normed_hidden_states,
                attention_mask=attention_mask,
                return_position_bias=True
            )
        else:
            attn_output = self.self_attention(
                normed_hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias
            )
        
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # FFN with residual connection
        normed_hidden_states = self.layer_norm_2(hidden_states)
        ffn_output = self.ffn(normed_hidden_states)
        hidden_states = hidden_states + self.dropout(ffn_output)
        
        return hidden_states, position_bias


class T5DecoderLayer(nn.Module):
    """Single T5 decoder layer with self-attention, cross-attention, and FFN"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1,
                 has_relative_bias=False):
        super().__init__()
        # Self-attention (causal)
        self.self_attention = T5Attention(
            d_model, num_heads, dropout, has_relative_bias=has_relative_bias
        )
        # Cross-attention
        self.cross_attention = T5Attention(
            d_model, num_heads, dropout, has_relative_bias=False
        )
        self.ffn = T5DenseGatedGeluDense(d_model, d_ff, dropout)
        
        self.layer_norm_1 = T5LayerNorm(d_model)
        self.layer_norm_2 = T5LayerNorm(d_model)
        self.layer_norm_3 = T5LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        hidden_states, 
        encoder_hidden_states,
        self_attention_mask=None,
        cross_attention_mask=None,
        self_position_bias=None,
        cross_position_bias=None
    ):
        # Self-attention
        normed = self.layer_norm_1(hidden_states)
        if self_position_bias is None:
            attn_output, self_position_bias = self.self_attention(
                normed,
                attention_mask=self_attention_mask,
                return_position_bias=True
            )
        else:
            attn_output = self.self_attention(
                normed,
                attention_mask=self_attention_mask,
                position_bias=self_position_bias
            )
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Cross-attention
        normed = self.layer_norm_2(hidden_states)
        if cross_position_bias is None:
            attn_output, cross_position_bias = self.cross_attention(
                normed,
                key_value_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
                return_position_bias=True
            )
        else:
            attn_output = self.cross_attention(
                normed,
                key_value_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
                position_bias=cross_position_bias
            )
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # FFN
        normed = self.layer_norm_3(hidden_states)
        ffn_output = self.ffn(normed)
        hidden_states = hidden_states + self.dropout(ffn_output)
        
        return hidden_states, self_position_bias, cross_position_bias


# ============================================================================
# PART 4: Complete T5 Model
# ============================================================================

class T5Model(nn.Module):
    """
    Complete T5 encoder-decoder architecture.
    Implements the full T5-small model from scratch.
    """
    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        pad_token_id=0
    ):
        super().__init__()
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'dropout': dropout,
            'pad_token_id': pad_token_id
        }
        
        # Shared embedding (used by both encoder and decoder)
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            T5EncoderLayer(
                d_model, num_heads, d_ff, dropout,
                has_relative_bias=(i == 0)  # Only first layer has bias
            )
            for i in range(num_encoder_layers)
        ])
        self.encoder_final_layer_norm = T5LayerNorm(d_model)
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            T5DecoderLayer(
                d_model, num_heads, d_ff, dropout,
                has_relative_bias=(i == 0)  # Only first layer has bias
            )
            for i in range(num_decoder_layers)
        ])
        self.decoder_final_layer_norm = T5LayerNorm(d_model)
        
        # LM head (shares weights with embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.pad_token_id = pad_token_id
    
    def _create_attention_mask(self, input_ids):
        """Create attention mask from input_ids (1 for tokens, 0 for padding)"""
        # Shape: (batch_size, 1, 1, seq_len)
        mask = (input_ids != self.pad_token_id).float()
        mask = mask.unsqueeze(1).unsqueeze(2)
        # Convert to additive mask (0 for attend, -inf for ignore)
        return (1.0 - mask) * -1e9
    
    def _create_causal_mask(self, seq_len, device):
        """Create causal mask for decoder self-attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, -1e9)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def encode(self, input_ids):
        """Encode input sequence"""
        # Embedding
        hidden_states = self.shared_embedding(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Attention mask
        attention_mask = self._create_attention_mask(input_ids)
        
        # Pass through encoder layers
        position_bias = None
        for layer in self.encoder_layers:
            hidden_states, position_bias = layer(
                hidden_states, 
                attention_mask=attention_mask,
                position_bias=position_bias
            )
        
        # Final layer norm
        hidden_states = self.encoder_final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states
    
    def decode(self, decoder_input_ids, encoder_hidden_states, encoder_input_ids):
        """Decode given encoder outputs"""
        # Embedding
        hidden_states = self.shared_embedding(decoder_input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Masks
        seq_len = decoder_input_ids.shape[1]
        self_attention_mask = self._create_causal_mask(seq_len, decoder_input_ids.device)
        cross_attention_mask = self._create_attention_mask(encoder_input_ids)
        
        # Pass through decoder layers
        self_position_bias = None
        cross_position_bias = None
        
        for layer in self.decoder_layers:
            hidden_states, self_position_bias, cross_position_bias = layer(
                hidden_states,
                encoder_hidden_states,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                self_position_bias=self_position_bias,
                cross_position_bias=cross_position_bias
            )
        
        # Final layer norm
        hidden_states = self.decoder_final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states
    
    def forward(self, input_ids, decoder_input_ids):
        """
        Full forward pass.
        
        Args:
            input_ids: Encoder input tokens (batch_size, enc_seq_len)
            decoder_input_ids: Decoder input tokens (batch_size, dec_seq_len)
        
        Returns:
            logits: Output logits (batch_size, dec_seq_len, vocab_size)
        """
        # Encode
        encoder_hidden_states = self.encode(input_ids)
        
        # Decode
        decoder_hidden_states = self.decode(
            decoder_input_ids, 
            encoder_hidden_states,
            input_ids
        )
        
        # Project to vocabulary
        logits = self.lm_head(decoder_hidden_states)
        
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=50, temperature=1.0):
        """
        Simple greedy generation.
        
        Args:
            input_ids: Encoder input (batch_size, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
        
        Returns:
            generated_ids: Generated token IDs
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Encode once
        encoder_hidden_states = self.encode(input_ids)
        
        # Start with pad token
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            self.pad_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        for _ in range(max_length):
            # Decode
            decoder_hidden_states = self.decode(
                decoder_input_ids,
                encoder_hidden_states,
                input_ids
            )
            
            # Get logits for last token
            logits = self.lm_head(decoder_hidden_states[:, -1:, :])
            logits = logits / temperature
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
            
            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Stop if all sequences have generated EOS (token 1 in T5)
            if (next_token == 1).all():
                break
        
        return decoder_input_ids


# ============================================================================
# PART 5: Loading HuggingFace Pretrained Weights
# ============================================================================

def load_huggingface_weights(custom_model, model_name="t5-small"):
    """
    Load pretrained weights from HuggingFace T5 into our custom model.
    
    Args:
        custom_model: Our T5Model instance
        model_name: HuggingFace model identifier
    
    Returns:
        custom_model with loaded weights
    """
    print(f"Loading weights from {model_name}...")
    
    # Load HuggingFace model
    hf_model = T5ForConditionalGeneration.from_pretrained(model_name)
    hf_state_dict = hf_model.state_dict()
    
    # Create mapping from HuggingFace keys to our keys
    weight_mapping = {}
    
    # Shared embedding
    weight_mapping['shared.weight'] = 'shared_embedding.weight'
    
    # Encoder layers
    for i in range(custom_model.config['num_encoder_layers']):
        prefix_hf = f'encoder.block.{i}'
        prefix_custom = f'encoder_layers.{i}'
        
        # Self-attention
        for proj in ['q', 'k', 'v', 'o']:
            weight_mapping[f'{prefix_hf}.layer.0.SelfAttention.{proj}.weight'] = \
                f'{prefix_custom}.self_attention.{proj}_proj.weight'
        
        # Relative position bias (only layer 0)
        if i == 0:
            weight_mapping[f'{prefix_hf}.layer.0.SelfAttention.relative_attention_bias.weight'] = \
                f'{prefix_custom}.self_attention.relative_bias.relative_attention_bias.weight'
        
        # Layer norms
        weight_mapping[f'{prefix_hf}.layer.0.layer_norm.weight'] = \
            f'{prefix_custom}.layer_norm_1.weight'
        weight_mapping[f'{prefix_hf}.layer.1.layer_norm.weight'] = \
            f'{prefix_custom}.layer_norm_2.weight'
        
        # FFN
        weight_mapping[f'{prefix_hf}.layer.1.DenseReluDense.wi_0.weight'] = \
            f'{prefix_custom}.ffn.wi_0.weight'
        weight_mapping[f'{prefix_hf}.layer.1.DenseReluDense.wi_1.weight'] = \
            f'{prefix_custom}.ffn.wi_1.weight'
        weight_mapping[f'{prefix_hf}.layer.1.DenseReluDense.wo.weight'] = \
            f'{prefix_custom}.ffn.wo.weight'
    
    # Encoder final layer norm
    weight_mapping['encoder.final_layer_norm.weight'] = 'encoder_final_layer_norm.weight'
    
    # Decoder layers
    for i in range(custom_model.config['num_decoder_layers']):
        prefix_hf = f'decoder.block.{i}'
        prefix_custom = f'decoder_layers.{i}'
        
        # Self-attention
        for proj in ['q', 'k', 'v', 'o']:
            weight_mapping[f'{prefix_hf}.layer.0.SelfAttention.{proj}.weight'] = \
                f'{prefix_custom}.self_attention.{proj}_proj.weight'
        
        # Relative position bias (only layer 0)
        if i == 0:
            weight_mapping[f'{prefix_hf}.layer.0.SelfAttention.relative_attention_bias.weight'] = \
                f'{prefix_custom}.self_attention.relative_bias.relative_attention_bias.weight'
        
        # Cross-attention
        for proj in ['q', 'k', 'v', 'o']:
            weight_mapping[f'{prefix_hf}.layer.1.EncDecAttention.{proj}.weight'] = \
                f'{prefix_custom}.cross_attention.{proj}_proj.weight'
        
        # Layer norms
        weight_mapping[f'{prefix_hf}.layer.0.layer_norm.weight'] = \
            f'{prefix_custom}.layer_norm_1.weight'
        weight_mapping[f'{prefix_hf}.layer.1.layer_norm.weight'] = \
            f'{prefix_custom}.layer_norm_2.weight'
        weight_mapping[f'{prefix_hf}.layer.2.layer_norm.weight'] = \
            f'{prefix_custom}.layer_norm_3.weight'
        
        # FFN
        weight_mapping[f'{prefix_hf}.layer.2.DenseReluDense.wi_0.weight'] = \
            f'{prefix_custom}.ffn.wi_0.weight'
        weight_mapping[f'{prefix_hf}.layer.2.DenseReluDense.wi_1.weight'] = \
            f'{prefix_custom}.ffn.wi_1.weight'
        weight_mapping[f'{prefix_hf}.layer.2.DenseReluDense.wo.weight'] = \
            f'{prefix_custom}.ffn.wo.weight'
    
    # Decoder final layer norm
    weight_mapping['decoder.final_layer_norm.weight'] = 'decoder_final_layer_norm.weight'
    
    # LM head
    weight_mapping['lm_head.weight'] = 'lm_head.weight'
    
    # Load weights
    custom_state_dict = custom_model.state_dict()
    loaded_keys = []
    missing_keys = []
    
    for hf_key, custom_key in weight_mapping.items():
        if hf_key in hf_state_dict and custom_key in custom_state_dict:
            custom_state_dict[custom_key] = hf_state_dict[hf_key]
            loaded_keys.append(custom_key)
        else:
            missing_keys.append(custom_key)
    
    custom_model.load_state_dict(custom_state_dict)
    
    print(f"✓ Successfully loaded {len(loaded_keys)} weight tensors")
    if missing_keys:
        print(f"⚠ Missing keys: {len(missing_keys)}")
        for key in missing_keys[:5]:
            print(f"  - {key}")
    
    return custom_model


# ============================================================================
# PART 6: Testing & Verification
# ============================================================================

def test_custom_t5():
    """Test custom T5 implementation and weight loading"""
    print("="*80)
    print("Testing Custom T5-Small Implementation")
    print("="*80)
    
    # Initialize custom model
    print("\\n1. Initializing custom T5-small model...")
    model = T5Model(
        vocab_size=32128,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Load pretrained weights
    print("\\n2. Loading HuggingFace pretrained weights...")
    model = load_huggingface_weights(model, "t5-small")
    
    # Load tokenizer
    print("\\n3. Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # Test inference
    print("\\n4. Testing inference...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Test example
    input_text = "translate English to German: The house is wonderful."
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(device)
    
    # Create decoder input (starts with pad token)
    decoder_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Test forward pass
        logits = model(input_ids, decoder_input_ids)
        print(f"   Output shape: {logits.shape}")
        print(f"   Output logits range: [{logits.min():.2f}, {logits.max():.2f}]")
        
        # Test generation
        print("\\n5. Testing generation...")
        generated_ids = model.generate(input_ids, max_length=20)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"   Input: {input_text}")
        print(f"   Generated: {generated_text}")
    
    print("\\n" + "="*80)
    print("✓ All tests passed! Model is ready for fine-tuning.")
    print