# imports
from topodoe import GrnCollection
from topodoe.topological_analysis import get_DVI

GRNs = GrnCollection.read('/home/mbouvier/git/DoE/article_DoE_for_GRN_selection_and_refinement/data/GRNs.h5')

dvi = get_DVI(GRNs)
breakpoint()