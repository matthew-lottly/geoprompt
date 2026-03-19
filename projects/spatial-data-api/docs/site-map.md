# Site Map

This repo has two reviewer-facing surfaces plus the typed API underneath them.

```mermaid
flowchart TD
    Root[/\//] --> Dashboard[/dashboard/]
    Root --> Docs[/docs/]
    Root --> Health[/health/]
    Root --> Ready[/health/ready/]

    Docs --> Metadata[/api/v1/metadata/]
    Docs --> Features[/api/v1/features/]
    Docs --> FeatureSummary[/api/v1/features/summary/]
    Docs --> OperationsSummary[/api/v1/summary/]
    Docs --> RecentObservations[/api/v1/observations/recent/]
    Docs --> ObservationExport[/api/v1/observations/export/]

    Features --> FeatureDetail[/api/v1/features/{feature_id}/]
    FeatureDetail --> FeatureObservations[/api/v1/features/{feature_id}/observations/]
    FeatureDetail --> ThresholdUpdate[/api/v1/stations/{feature_id}/thresholds/]
```

## Reading Order

1. Start with the generated monitoring-status map in the README.
2. Open `/dashboard` for the human-facing monitoring view.
3. Open `/docs` for the typed API contract.
4. Use the feature and summary endpoints for data review.
