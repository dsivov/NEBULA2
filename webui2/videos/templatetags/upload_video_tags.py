from django import template

register = template.Library()


status_to_name = {
    'initial': 'Start',
    'uploading': 'Uploading',
}


@register.filter
def get_vpm_status_name(status):
    return status_to_name.get(status, status)
