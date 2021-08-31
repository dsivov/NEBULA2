import json

from django import forms
from django.core.validators import FileExtensionValidator

from nebula_api.search_api import get_similarity_algorithms_available


CAMERA_CHOICES = (
    (
        {"hfov": 129, "sheight": 4.04, "fheight": 1080, "flength": 5},
        'Xiaomi YI'
    ),
    (
        {"hfov": 180, "sheight": 3.6, "fheight": 1440, "flength": 5},
        'Garmin 66W'
    ),
    (
        {"hfov": 170, "sheight": 4.04, "fheight": 1080, "flength": 5},
        'Aukey DR01'
    )
)

LOG_CHOICES = (
    (False, 'No'),
    (True, 'Yes')
)

UPLOAD_FROM_CHOICES = (
    ('filesystem', 'New Video'),
    ('s3', 'Video/s from s3')
)


class VideoForm(forms.Form):
    upload_from = forms.ChoiceField(
        widget=forms.RadioSelect,
        choices=UPLOAD_FROM_CHOICES,
        required=True,
        initial='filesystem'
    )
    video = forms.FileField(
        required=False,
        validators=[FileExtensionValidator(allowed_extensions=['avi', 'mp4'])],
        widget=forms.FileInput(
            attrs={
                'class': "filestyle",
                'data-text': "Find file",
                'data-btnClass': "btn btn-dark",
            }
        )
    )
    metafields = forms.CharField(
        widget=forms.Textarea(
            attrs={'rows': 5, 'cols': 24, 'placeholder': 'Enter video metadata here'}
        ),
        required=False
    )
#    videofile = forms.FileField(upload_to='media/videos/')
    segment = forms.DecimalField(required=True, label='Segment time(s)',
                                 max_digits=3,
                                 decimal_places=0, initial=5)
#    focal_length =  forms.DecimalField(required=True,
#                                       label='focal length (mm)',
#                                       max_digits=2)
#    frame_hight =  forms.DecimalField(required=True,
#                                      label='frame hight (px)',
#                                      max_digits=2)
#    sensor_hight = forms.DecimalField(required=True,
#                                      label='sensor hight (mm)',
#                                      max_digits=2)
#
#    camera_type = forms.ChoiceField(choices=CAMERA_CHOICES,
#                                    widget=forms.Select)

    log_data = forms.TypedChoiceField(label='Log Tagging?',
                                      choices=LOG_CHOICES,
                                      widget=forms.RadioSelect,
                                      required=True,
                                      initial=False)
    choose_video_from_s3 = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        label='',
        required=False,
        choices=()
    )

    def __init__(self, available_filenames_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['choose_video_from_s3'].choices = zip(
            available_filenames_list,
            available_filenames_list
        )
        json_available_filenames_list = json.dumps(
            available_filenames_list
        )
        self.fields['available_filenames_list'] = forms.CharField(
            max_length=len(json_available_filenames_list),
            initial=json_available_filenames_list,
            required=True,
            widget=forms.HiddenInput(),
        )

    def clean(self):
        if (
            self.cleaned_data['upload_from'] == 'filesystem'
            and 'video' in self.cleaned_data
            and not self.cleaned_data['video']
        ):
            self.add_error('video', 'Video not chosen')
        if (
            self.cleaned_data['upload_from'] == 's3'
            and 'choose_video_from_s3' in self.cleaned_data
            and not self.cleaned_data['choose_video_from_s3']
        ):
            self.add_error('choose_video_from_s3', 'Video/s not chosen')


class VideoSearchForm(forms.Form):
    search = forms.CharField(
        widget=forms.TextInput(
            attrs={
                'placeholder': 'Search',
                'style': 'font-size: 0.9rem;',
                'id': 'search-form-search-input'
            }
        ),
        required=False
    )
    size = forms.TypedChoiceField(
        widget=forms.Select(
            attrs={
                'onchange': "onChangePageSize()",
                'class': 'form-select',
                'style': 'min-width:80px;font-size: 0.9rem;',
                'id': 'search-form-size-input'
            }
        ),
        coerce=int,
        choices=[(i, i) for i in (10, 25, 50, 100, 250, 500)],
        initial=10,
        required=False
    )
    search_method = forms.ChoiceField(
        widget=forms.Select(
            attrs={
                'class': 'form-select',
                'style': 'min-width:80px;font-size: 0.9rem;',
                'id': 'search-form-search-method-input'
            }
        ),
        choices=(('el_search', 'Elastic Search'), ('bert2clip', 'Text Query - BERT'), ('doc2vec', 'Text Query - Doc2Vec')),
        required=False
    )
    page = forms.IntegerField(
        widget=forms.NumberInput(
            attrs={
                'class': 'mx-2 form-control',
                'style': 'width:60px;font-size: 0.9rem;',
                'type': 'number',
                'value': '1',
                'min': '1',
                'id': 'search-form-page-input'
            }
        ),
        required=False
    )
    sort_by = forms.ChoiceField(
        widget=forms.HiddenInput(
            attrs={
                'id': 'search-form-sort-by'
            }
        ),
        choices=(("name", "name"), ("timestamp", "timestamp")),
        required=False
    )
    sort_direction = forms.ChoiceField(
        widget=forms.HiddenInput(
            attrs={
                'id': 'search-form-sort-direction'
            }
        ),
        choices=(("asc", "asc"), ("desc", "desc")),
        required=False
    )
    SimilarID = forms.CharField(
        widget=forms.HiddenInput(
            attrs={
                'id': 'similar-id-input',
                'class': 'form-select',
                'style': 'min-width:80px;font-size: 0.9rem;'
            }
        ),
        required=False
    )
    similarity_algo = forms.ChoiceField(
        widget=forms.Select(
            attrs={
                'class': 'form-select',
                'style': 'min-width:80px;font-size: 0.9rem;',
                'id': 'search-form-similarity-algo'
            }
        ),
        choices=get_similarity_algorithms_available,
        required=False
    )


class VideoAnnotateForm(forms.Form):
    annotate = forms.CharField(widget=forms.HiddenInput())


datasets = [
    ('CM', 'Hollywood2'),
    ('TR', 'Hollywood3'),
    ('100', 'Hollywood4'),
    ('205', 'LSMDC'),
    ('VRT', 'Clips'),
    ('DD', 'Clips Movie'),
]

databases = [
    ('Dev', 'nebula_dev'),
    ('Hollywood2_old', 'nebula_hollywood'),
    ('Hollywood2_full', 'nebula_datadriven'),
]

es_indeses = [
    ('Dev', 'dev'),
    ('Hollywood2_old', 'nebula_hollywood'),
    ('Hollywood2_full', 'datadriven'),
]

class SettingsChoiceForm(forms.Form):
    user_settings = forms.ChoiceField(
        widget=forms.Select(
            attrs={
                'class': 'form-select',
                'style': 'min-width:80px;font-size: 0.9rem;'
            }
        ),
        choices=[(code, text) for code, text in datasets])
    
    data_base = forms.ChoiceField(
        widget=forms.Select(
            attrs={
                'class': 'form-select',
                'style': 'min-width:80px;font-size: 0.9rem;'
            }
        ),
        choices=[(code, text) for code, text in databases])
    
    es_index = forms.ChoiceField(
        widget=forms.Select(
            attrs={
                'class': 'form-select',
                'style': 'min-width:80px;font-size: 0.9rem;'
            }
        ),
        choices=[(code, text) for code, text in es_indeses])


