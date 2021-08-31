from django import template
from django.template.defaultfilters import stringfilter
import re

register = template.Library()


@register.filter
def url_for_id(url):
    return re.sub('[/:?=]', '', url)
