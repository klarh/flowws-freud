from flowws import try_to_import

from .version import __version__

LocalDensity = try_to_import('.LocalDensity', 'LocalDensity', __name__)
RDF = try_to_import('.RDF', 'RDF', __name__)
SmoothBOD = try_to_import('.SmoothBOD', 'SmoothBOD', __name__)
Steinhardt = try_to_import('.Steinhardt', 'Steinhardt', __name__)
