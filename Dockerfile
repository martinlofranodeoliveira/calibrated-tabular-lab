ARG DEPENDENCY_IMAGE=calibrated-tabular-dependencies:local
FROM ${DEPENDENCY_IMAGE}
COPY --chown=1000:1000 . /opt/project
WORKDIR /work
USER 1000:1000
CMD ["sh", "/opt/project/scripts/offline-test.sh"]
