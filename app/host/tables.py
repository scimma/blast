import django_tables2 as tables
from django.db.models.functions import Coalesce
from django.db.models import F

from .models import Transient


class TransientTable(tables.Table):
    name = tables.TemplateColumn(
        "<a href=\"{% url 'results' record.name %}\">{{ record.get_display_name }}</a>",
        verbose_name="Name",
        orderable=True,
        order_by="name",
    )

    # prefix = tables.Column(
    #    accessor="tns_prefix", verbose_name="Prefix", orderable=True, order_by="prefix"
    # )

    disc_date = tables.Column(
        accessor="public_timestamp",
        verbose_name="Discovery Date",
        orderable=True,
        order_by="public_timestamp",
    )

    def order_disc_date(self, queryset, is_descending):
        # This solution was adapted from https://bleepingcoder.com/django-tables2/346842660/allow-descending-sort-columns-to-place-empty-values-at  # noqa
        if is_descending:
            queryset = queryset.order_by(F('public_timestamp').desc(nulls_last=True))
        else:
            queryset = queryset.order_by(F('public_timestamp').asc(nulls_last=True))
        return (queryset, True)

    ra_string = tables.Column(
        accessor="ra", verbose_name="Right Ascension", orderable=True, order_by="ra_deg"
    )

    dec_string = tables.Column(
        accessor="dec", verbose_name="Declination", orderable=True, order_by="dec_deg"
    )

    spec_type = tables.Column(
        accessor="spectroscopic_class",
        verbose_name="Spec. Class",
        orderable=True,
        order_by="spectroscopic_class",
    )

    best_spec_redshift = tables.TemplateColumn(
        "{{ record.best_spec_redshift|floatformat:3 }}",
        accessor="best_spec_redshift",
        verbose_name="Redshift",
        orderable=True,
        order_by="host__redshift",
    )

    progress = tables.TemplateColumn(
        """          {% if record.progress == 0 %}
              Waiting
          {% elif record.progress == 100 %}
            <div class="progress">
              <div class="progress-bar bg-success" role="progressbar" style="width: {{record.progress}}%;" aria-valuenow="{{transient.progress}}" aria-valuemin="0" aria-valuemax="100">{{record.progress}}%</div>
            </div>
          {% else %}
            <div class="progress">
              <div class="progress-bar" role="progressbar" style="width: {{record.progress}}%;" aria-valuenow="{{record.progress}}" aria-valuemin="0" aria-valuemax="100">{{record.progress}}%</div>
            </div>
          {% endif %}
""",
        verbose_name="Progress",
        orderable=True,
        order_by="progress",
    )

    class Meta:
        model = Transient
        fields = ()

        template_name = "django_tables2/bootstrap4.html"
        attrs = {
            "th": {
                "_ordering": {
                    "orderable": "sortable",  # Instead of `orderable`
                    "ascending": "ascend",  # Instead of `asc`
                    "descending": "descend",  # Instead of `desc`
                }
            },
            "class": "table table-bordered table-hover",
            "id": "k2_transient_tbl",
            "columnDefs": [
                {"type": "title-numeric", "targets": 1},
                {"type": "title-numeric", "targets": 2},
            ],
            "order": [[3, "desc"]],
        }

    def order_best_spec_redshift(self, queryset, is_descending):
        queryset = queryset.annotate(
            best_spec_redshift=Coalesce("redshift", "host__redshift"),
        ).order_by(("-" if is_descending else "") + "best_spec_redshift")

        return (queryset, True)
