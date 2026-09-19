FROM grafana/grafana:latest

COPY provisioning/ /etc/grafana/provisioning/

COPY dashboards/dashboards.yml /etc/grafana/provisioning/dashboards/

COPY dashboards/nexusml_main.json /var/lib/grafana/dashboards/