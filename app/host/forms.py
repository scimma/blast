from django import forms


class TransientSearchForm(forms.Form):
    name = forms.CharField(
        label="",
        widget=forms.TextInput(
            attrs={"placeholder": "e.g. 2022eqw", "style": "width:10em"}
        ),
        required=False,
    )

    # optional "status" is read by the transient_list view
    status = forms.CharField(label="", initial="all", required=False)


class ImageGetForm(forms.Form):
    def __init__(self, *args, **kwargs):
        filter_choices = kwargs.pop("filter_choices")
        super(ImageGetForm, self).__init__(*args, **kwargs)
        choices = [(filter, filter) for filter in filter_choices]
        choices.insert(0, (None, "Choose cutout"))
        self.fields["filters"] = forms.ChoiceField(
            label="",
            choices=choices,
            widget=forms.Select(attrs={"placeholder": "select cutout"}),
        )


class TransientUploadForm(forms.Form):
    tns_names = forms.CharField(
        widget=forms.Textarea,
        label="Transients by Name, using TNS to gather additional information",
        required=False,
    )
    retrigger = forms.BooleanField(
        widget=forms.CheckboxInput,
        label="Check this box to retrigger workflows of existing transients in the list "
        "instead of ignoring them. This is useful for resuming workflows that have been "
        "prematurely aborted; completed tasks in the workflows will not be reprocessed.",
        required=False,
    )
    full_info = forms.CharField(
        widget=forms.Textarea,
        label='Comma-separated: Name, RA, Dec, Redshift, Classification.  RA/Dec must be '
        'decimal degrees and use "None" to indicate missing redshift or classification.',
        required=False,
    )
