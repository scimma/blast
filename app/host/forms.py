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
        widget=forms.Textarea(attrs={
            'style': 'min-width: 30rem;',
            'cols': 100,
            'placeholder': '2024abc\n2025xyz'
        }),
        label="List of TNS transient names (one per line)",
        required=False,
    )
    full_info = forms.CharField(
        widget=forms.Textarea(attrs={
            'style': 'min-width: 30rem;',
            'cols': 100,
            'placeholder': 'Identifier, RA, Dec, Redshift, Classification, Display Name'
        }),
        label='Transient definition table',
        required=False,
    )
