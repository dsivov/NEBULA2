from django import template
import re
from urllib.parse import quote_plus
from config.config import NEBULA_CONF

register = template.Library()
conf = NEBULA_CONF()


@register.filter
def url_for_id(url):
    return re.sub('[/:?=]', '', url)


@register.simple_tag
def get_arangodb_graph_view_url(movie_id):
    node_start = f'nodeStart={quote_plus(movie_id)}'
    path = '_db/nebula_development/_admin/aardvark/index.html#graph/StoryGraph?nodeLabelByCollection=false&nodeColorByCollection=true&nodeSizeByEdges=true&edgeLabelByCollection=false&edgeColorByCollection=false&depth=200&limit=250&nodeLabel=description&nodeColor=%232ecc71&nodeColorAttribute=&nodeSize=&edgeLabel=&edgeColor=%23cccccc&edgeColorAttribute=&edgeEditable=true'
    return f"http://{conf.DEFAULT_ARANGO_USER}:{conf.DEFAULT_ARANGO_PASSWORD}@{conf.get_arango_graphs_host()}/{path}&{node_start}"


@register.simple_tag
def get_arangodb_graph_view_path():
    path = '_db/nebula_development/_admin/aardvark/index.html#graph/StoryGraph?nodeLabelByCollection=false&nodeColorByCollection=true&nodeSizeByEdges=true&edgeLabelByCollection=false&edgeColorByCollection=false&depth=200&limit=250&nodeLabel=description&nodeColor=%232ecc71&nodeColorAttribute=&nodeSize=&edgeLabel=&edgeColor=%23cccccc&edgeColorAttribute=&edgeEditable=true'
    return f"http://{conf.DEFAULT_ARANGO_USER}:{conf.DEFAULT_ARANGO_PASSWORD}@{conf.get_arango_graphs_host()}/{path}"
