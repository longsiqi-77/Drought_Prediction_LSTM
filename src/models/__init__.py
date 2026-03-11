from .lstm import LSTMModel
from .transformer import SimpleTransformer, TimeSeriesTransformer

def create_model(model_type, input_size, prediction_steps, **kwargs):
    model_type = model_type.lower()
    
    if model_type == "lstm":
        return LSTMModel(
            input_size=input_size,
            hidden_size=kwargs.get("hidden_size", 64),
            num_layers=kwargs.get("num_layers", 2),
            output_size=1,
            prediction_steps=prediction_steps,
            dropout=kwargs.get("dropout", 0.2)
        )
    
    elif model_type == "transformer":
        return TimeSeriesTransformer(
            input_size=input_size,
            d_model=kwargs.get("d_model", 64),
            nhead=kwargs.get("nhead", 4),
            num_encoder_layers=kwargs.get("num_encoder_layers", 2),
            num_decoder_layers=kwargs.get("num_decoder_layers", 2),
            dim_feedforward=kwargs.get("dim_feedforward", 256),
            prediction_steps=prediction_steps,
            dropout=kwargs.get("dropout", 0.1)
        )
    
    elif model_type == "simpletransformer":
        return SimpleTransformer(
            input_size=input_size,
            d_model=kwargs.get("d_model", 64),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", 2),
            prediction_steps=prediction_steps,
            dropout=kwargs.get("dropout", 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: [LSTM, Transformer, SimpleTransformer]")

__all__ = ["LSTMModel", "TimeSeriesTransformer", "SimpleTransformer", "create_model"]