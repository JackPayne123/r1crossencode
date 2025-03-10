from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator, Literal

from dataclasses_json import dataclass_json
from rich import print as rprint
from rich.table import Table
from rich.tree import Tree

SEQUENCES_CONFIG_HELP = dict(
    buffer="How many tokens to add as context to each sequence, on each side. The tokens chosen for the top acts / \
quantile groups can't be outside the buffer range. If None, we use the entire sequence as context.",
    compute_buffer="If False, then we don't compute the loss effect, activations, or any other data for tokens \
other than the bold tokens in our sequences (saving time).",
    n_quantiles="Number of quantile groups for the sequences. If zero, we only show top activations, no quantile \
groups.",
    top_acts_group_size="Number of sequences in the 'top activating sequences' group.",
    quantile_group_size="Number of sequences in each of the sequence quantile groups.",
    top_logits_hoverdata="Number of top/bottom logits to show in the hoverdata for each token.",
    stack_mode="How to stack the sequence groups.\n  'stack-all' = all groups are stacked in a single column \
(scrolls vertically if it overflows)\n  'stack-quantiles' = first col contains top acts, second col contains all \
quantile groups\n  'stack-none' = we stack in a way which ensures no vertical scrolling.",
    hover_below="Whether the hover information about a token appears below or above the token.",
)

ACTIVATIONS_HISTOGRAM_CONFIG_HELP = dict(
    n_bins="Number of bins for the histogram.",
)

LOGITS_HISTOGRAM_CONFIG_HELP = dict(
    n_bins="Number of bins for the histogram.",
)

LOGITS_TABLE_CONFIG_HELP = dict(
    n_rows="Number of top/bottom logits to show in the table.",
)

FEATURE_TABLES_CONFIG_HELP = dict(
    n_rows="Number of rows to show for each feature table.",
    neuron_alignment_table="Whether to show the neuron alignment table.",
    correlated_neurons_table="Whether to show the correlated neurons table.",
    correlated_features_table="Whether to show the (pairwise) correlated features table.",
    correlated_b_features_table="Whether to show the correlated encoder-B features table.",
)


@dataclass
class BaseComponentConfig:
    def data_is_contained_in(self, other: "BaseComponentConfig") -> bool:
        """
        This returns False only when the data that was computed based on `other` wouldn't be enough to show the data
        that was computed based on `self`. For instance, if `self` was a config object with 10 rows, and `other` had
        just 5 rows, then this would return False. A less obvious example: if `self` was a histogram config with 50 bins
        then `other` would need to have exactly 50 bins (because we can't change the bins after generating them).
        """
        return True

    @property
    def help_dict(self) -> dict[str, str]:
        """
        This is a dictionary which maps the name of each argument to a description of what it does. This is used when
        printing out the help for a config object, to show what each argument does.
        """
        return {}


@dataclass
class PromptConfig(BaseComponentConfig):
    pass


@dataclass
class SequencesConfig(BaseComponentConfig):
    buffer: tuple[int, int] | None = (5, 5)
    compute_buffer: bool = True
    n_quantiles: int = 10
    top_acts_group_size: int = 20
    quantile_group_size: int = 5
    top_logits_hoverdata: int = 5
    stack_mode: Literal["stack-all", "stack-quantiles", "stack-none"] = "stack-all"
    hover_below: bool = True

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return all(
            [
                self.buffer is None
                or (
                    other.buffer is not None and self.buffer[0] <= other.buffer[0]
                ),  # the buffer needs to be <=
                self.buffer is None
                or (other.buffer is not None and self.buffer[1] <= other.buffer[1]),
                int(self.compute_buffer)
                <= int(
                    other.compute_buffer
                ),  # we can't compute the buffer if we didn't in `other`
                self.n_quantiles
                in {
                    0,
                    other.n_quantiles,
                },  # we actually need the quantiles identical (or one to be zero)
                self.top_acts_group_size
                <= other.top_acts_group_size,  # group size needs to be <=
                self.quantile_group_size
                <= other.quantile_group_size,  # each quantile group needs to be <=
                self.top_logits_hoverdata
                <= other.top_logits_hoverdata,  # hoverdata rows need to be <=
            ]
        )

    def __post_init__(self):
        # Get list of group lengths, based on the config params
        self.group_sizes = [self.top_acts_group_size] + [
            self.quantile_group_size
        ] * self.n_quantiles

    @property
    def help_dict(self) -> dict[str, str]:
        return SEQUENCES_CONFIG_HELP


@dataclass
class ActsHistogramConfig(BaseComponentConfig):
    n_bins: int = 50

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_bins == other.n_bins

    @property
    def help_dict(self) -> dict[str, str]:
        return ACTIVATIONS_HISTOGRAM_CONFIG_HELP


@dataclass
class LogitsHistogramConfig(BaseComponentConfig):
    n_bins: int = 50

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_bins == other.n_bins

    @property
    def help_dict(self) -> dict[str, str]:
        return LOGITS_HISTOGRAM_CONFIG_HELP


@dataclass
class LogitsTableAConfig(BaseComponentConfig):
    n_rows: int = 4

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_rows <= other.n_rows

    @property
    def help_dict(self) -> dict[str, str]:
        return LOGITS_TABLE_CONFIG_HELP

@dataclass
class LogitsTableBConfig(BaseComponentConfig):
    n_rows: int = 4

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_rows <= other.n_rows

    @property
    def help_dict(self) -> dict[str, str]:
        return LOGITS_TABLE_CONFIG_HELP


@dataclass
class FeatureTablesConfig(BaseComponentConfig):
    n_rows: int = 2
    neuron_alignment_table: bool = False
    relative_decoder_strength_table: bool = True
    decoder_cosine_sim_table: bool = True
    correlated_neurons_table: bool = False
    correlated_features_table: bool = False
    correlated_b_features_table: bool = False

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return all(
            [
                self.n_rows <= other.n_rows,
                self.neuron_alignment_table <= other.neuron_alignment_table,
                self.decoder_cosine_sim_table <= other.decoder_cosine_sim_table,
                self.relative_decoder_strength_table <= other.relative_decoder_strength_table,
                self.correlated_neurons_table <= other.correlated_neurons_table,
                self.correlated_features_table <= other.correlated_features_table,
                self.correlated_b_features_table <= other.correlated_b_features_table,
            ]
        )

    @property
    def help_dict(self) -> dict[str, str]:
        return FEATURE_TABLES_CONFIG_HELP


GenericComponentConfig = (
    PromptConfig
    | SequencesConfig
    | ActsHistogramConfig
    | LogitsHistogramConfig
    | LogitsTableAConfig
    | LogitsTableBConfig
    | FeatureTablesConfig
)


class Column:
    def __init__(
        self,
        *args: GenericComponentConfig,
        width: int | None = None,
    ):
        self.components = list(args)
        self.width = width

    def __iter__(self) -> Iterator[Any]:
        return iter(self.components)

    def __getitem__(self, idx: int) -> Any:
        return self.components[idx]

    def __len__(self) -> int:
        return len(self.components)


@dataclass_json
@dataclass
class SaeVisLayoutConfig:
    """
    This object allows you to set all the ways the feature vis will be laid out.

    Args (specified by the user):
        columns:
            A list of `Column` objects, where each `Column` contains a list of component configs.
        height:
            The height of the vis (in pixels).

    Args (defined during __init__):
        seq_cfg:
            The `SequencesConfig` object, which contains all the parameters for the top activating sequences (and the
            quantile groups).
        act_hist_cfg:
            The `ActsHistogramConfig` object, which contains all the parameters for the activations histogram.
        logits_hist_cfg:
            The `LogitsHistogramConfig` object, which contains all the parameters for the logits histogram.
        logits_table_cfg_A:
            The `LogitsTableAConfig` object, which contains all the parameters for the logits table A.
        logits_table_cfg_B:
            The `LogitsTableBConfig` object, which contains all the parameters for the logits table B.
        feature_tables_cfg:
            The `FeatureTablesConfig` object, which contains all the parameters for the feature tables.
        prompt_cfg:
            The `PromptConfig` object, which contains all the parameters for the prompt-centric vis.
    """

    columns: dict[int | tuple[int, int], Column] = field(default_factory=dict)
    height: int = 750

    seq_cfg: SequencesConfig | None = None
    act_hist_cfg: ActsHistogramConfig | None = None
    logits_hist_cfg: LogitsHistogramConfig | None = None
    logits_table_cfg_A: LogitsTableAConfig | None = None
    logits_table_cfg_B: LogitsTableBConfig | None = None
    feature_tables_cfg: FeatureTablesConfig | None = None
    prompt_cfg: PromptConfig | None = None

    def __init__(self, columns: list[Column], height: int = 750):
        """
        The __init__ method will allow you to extract things like `self.seq_cfg` from the object (even though they're
        initially stored in the `columns` attribute). It also verifies that there are no duplicate components (which is
        redundant, and could mess up the HTML).
        """
        # Define the columns (as dict) and the height
        self.columns = {idx: col for idx, col in enumerate(columns)}
        self.height = height

        # Get a list of all our components, and verify there's no duplicates
        all_components = [
            component for column in self.columns.values() for component in column
        ]
        all_component_names = [
            comp.__class__.__name__.rstrip("Config") for comp in all_components
        ]
        assert len(all_component_names) == len(
            set(all_component_names)
        ), "Duplicate components in layout config"
        self.components: dict[str, BaseComponentConfig] = {
            name: comp for name, comp in zip(all_component_names, all_components)
        }

        # Once we've verified this, store each config component as an attribute
        for comp, comp_name in zip(all_components, all_component_names):
            match comp_name:
                case "Prompt":
                    self.prompt_cfg = comp
                case "Sequences":
                    self.seq_cfg = comp
                case "ActsHistogram":
                    self.act_hist_cfg = comp
                case "LogitsHistogram":
                    self.logits_hist_cfg = comp
                case "LogitsTableA":
                    self.logits_table_cfg_A = comp
                case "LogitsTableB":
                    self.logits_table_cfg_B = comp
                case "FeatureTables":
                    self.feature_tables_cfg = comp
                case _:
                    raise ValueError(f"Unknown component name {comp_name}")

    def data_is_contained_in(self, other: "SaeVisLayoutConfig") -> bool:
        """
        Returns True if `self` uses only data that would already exist in `other`. This is useful because our prompt-
        centric vis needs to only use data that was already computed as part of our initial data gathering. For example,
        if our SaeVisData object only contains the first 10 rows of the logits table, then we can't show the top 15 rows
        in the prompt centric view!
        """
        for comp_name, comp in self.components.items():
            # If the component in `self` is not present in `other`, return False
            if comp_name not in other.components:
                return False
            # If the component in `self` is present in `other`, but the `self` component is larger, then return False
            comp_other = other.components[comp_name]
            if not comp.data_is_contained_in(comp_other):
                return False

        return True

    def help(
        self,
        title: str = "SaeVisLayoutConfig",
        key: bool = True,
    ) -> Tree | None:
        """
        This prints out a tree showing the layout of the vis, by column (as well as the values of the arguments for each
        config object, plus their default values if they changed, and the descriptions of each arg).
        """

        # Create tree (with title and optionally the key explaining arguments)
        if key:
            title += "\n\n" + KEY_LAYOUT_VIS
        tree = Tree(title)

        n_columns = len(self.columns)

        # For each column, add a tree node
        for column_idx, vis_components in self.columns.items():
            n_components = len(vis_components)
            tree_column = tree.add(f"Column {column_idx}")

            # For each component in that column, add a tree node
            for component_idx, vis_component in enumerate(vis_components):
                n_params = len(asdict(vis_component))
                tree_component = tree_column.add(
                    f"{vis_component.__class__.__name__}".rstrip("Config")
                )

                # For each config parameter of that component
                for param_idx, (param, value) in enumerate(
                    asdict(vis_component).items()
                ):
                    # Get line break if we're at the final parameter of this component (unless it's the final component
                    # in the final column)
                    suffix = "\n" if (param_idx == n_params - 1) else ""
                    if (component_idx == n_components - 1) and (
                        column_idx == n_columns - 1
                    ):
                        suffix = ""

                    # Get argument description, and its default value
                    desc = vis_component.help_dict.get(param, "")
                    value_default = getattr(
                        vis_component.__class__, param, "no default"
                    )

                    # Add tree node (appearance is different if value is changed from default)
                    if value != value_default:
                        info = f"[b dark_orange]{param}: {value!r}[/] ({value_default!r}) \n[i]{desc}[/i]{suffix}"
                    else:
                        info = (
                            f"[b #00aa00]{param}: {value!r}[/] \n[i]{desc}[/i]{suffix}"
                        )
                    tree_component.add(info)

        rprint(tree)

    @classmethod
    def default_feature_centric_layout(cls) -> "SaeVisLayoutConfig":
        return cls(
            columns=[
                Column(
                    ActsHistogramConfig(), 
                    LogitsTableAConfig(n_rows=4),
                    LogitsTableBConfig(n_rows=4),
                    FeatureTablesConfig()
                ),
                Column(SequencesConfig(stack_mode="stack-none")),
            ],
            height=750,
        )

    @classmethod
    def default_prompt_centric_layout(cls) -> "SaeVisLayoutConfig":
        return cls(
            columns=[
                Column(
                    PromptConfig(),
                    ActsHistogramConfig(),
                    LogitsTableAConfig(n_rows=4),
                    LogitsTableBConfig(n_rows=4),
                    SequencesConfig(top_acts_group_size=10, n_quantiles=0),
                    width=450,
                ),
            ],
            height=1000,
        )


KEY_LAYOUT_VIS = """Key: 
  the tree shows which components will be displayed in each column (from left to right)
  arguments are [b #00aa00]green[/]
  arguments changed from their default are [b dark_orange]orange[/], with default in brackets
  argument descriptions are in [i]italics[/i]
"""


SAE_CONFIG_DICT = dict(
    hook_point="The hook point to use for the SAE",
    features="The set of features which we'll be gathering data for. If an integer, we only get data for 1 feature",
    minibatch_size_tokens="The minibatch size we'll use to split up the full batch during forward passes, to avoid \
OOMs.",
    minibatch_size_features="The feature minibatch size we'll use to split up our features, to avoid OOM errors",
    seed="Random seed, for reproducibility (e.g. sampling quantiles)",
    verbose="Whether to print out progress messages and other info during the data gathering process",
)


@dataclass_json
@dataclass
class SaeVisConfig:
    # Data
    hook_point: str | None = None
    features: int | Iterable[int] | None = None
    minibatch_size_features: int = 256
    minibatch_size_tokens: int = 64

    # Vis
    feature_centric_layout: SaeVisLayoutConfig = field(
        default_factory=SaeVisLayoutConfig.default_feature_centric_layout
    )
    prompt_centric_layout: SaeVisLayoutConfig = field(
        default_factory=SaeVisLayoutConfig.default_prompt_centric_layout
    )

    # Misc
    seed: int | None = 0
    verbose: bool = False

    # Depreciated
    batch_size: None = None

    def __post_init__(self):
        assert (
            self.batch_size is None
        ), "The `batch_size` parameter has been depreciated. Please use `minibatch_size_tokens` instead."

    def to_dict(self) -> dict[str, Any]:
        """Used for type hinting (the actual method comes from the `dataclass_json` decorator)."""
        ...

    def help(self, title: str = "SaeVisConfig"):
        """
        Performs the `help` method for both of the layout objects, as well as for the non-layout-based configs.
        """
        # Create table for all the non-layout-based params
        table = Table(
            "Param", "Value (default)", "Description", title=title, show_lines=True
        )

        # Populate table (middle row is formatted based on whether value has changed from default)
        for param, desc in SAE_CONFIG_DICT.items():
            value = getattr(self, param)
            value_default = getattr(self.__class__, param, "no default")
            if value != value_default:
                value_default_repr = (
                    "no default"
                    if value_default == "no default"
                    else repr(value_default)
                )
                value_str = f"[b dark_orange]{value!r}[/]\n({value_default_repr})"
            else:
                value_str = f"[b #00aa00]{value!r}[/]"
            table.add_row(param, value_str, f"[i]{desc}[/]")

        # Print table, and print the help trees for the layout objects
        rprint(table)
        self.feature_centric_layout.help(
            title="SaeVisLayoutConfig: feature-centric vis", key=False
        )
        self.prompt_centric_layout.help(
            title="SaeVisLayoutConfig: prompt-centric vis", key=False
        )
