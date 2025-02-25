from .adapter import CrossCoderAdapter, CrossCoderConfig
from .transformer_lens_adapter import TransformerLensWrapperAdapter
from .observable_model import ObservableModel
from .utils import (
    get_model_hook_points, 
    discover_hook_point_for_layer, 
    load_crosscoder_checkpoint,
    setup_visualization_data,
    save_feature_data,
    get_feature_importance
)
from .visualization import (
    import_sae_vis,
    prepare_feature_visualization,
    visualize_top_features
)
# Import the new sae_vis compatibility layer
from .sae_vis_compat import (
    DirectCrossCoderAdapter,
    ModelWrapper,
    create_sae_vis_compatible_encoder,
    create_model_wrapper,
    setup_sae_vis_data,
    import_sae_vis_modules
)

__version__ = "0.1.0" 