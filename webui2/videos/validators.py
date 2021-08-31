from django.core.exceptions import ValidationError

def validate_video(value):

    value = str(value)
    print(value)
    if value.endswith(".mp4") != True and value.endswith(".webm") != True and value.endswith(".ogg") != True:
        raise ValidationError("The video file must be .mp4, .webm, or .ogg")
    else:
        return value
