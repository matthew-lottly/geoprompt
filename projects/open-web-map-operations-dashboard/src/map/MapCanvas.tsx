import type { DashboardLayer } from "./types";


interface MapCanvasProps {
  layers: DashboardLayer[];
}


export function MapCanvas({ layers }: MapCanvasProps) {
  return (
    <section className="map-panel">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Map Surface</p>
          <h2>Open Operations View</h2>
        </div>
        <p className="section-copy">Conceptual MapLibre/OpenLayers-style review surface for operational layers and feature inspection.</p>
      </div>
      <div className="map-surface" aria-label="Operations map concept">
        <div className="map-region west">West</div>
        <div className="map-region central">Central</div>
        <div className="map-region east">East</div>
        {layers.map((layer) => (
          <div
            key={layer.id}
            className={`map-marker ${layer.status}`}
            style={{ left: `${layer.center.x}%`, top: `${layer.center.y}%` }}
            aria-label={layer.title}
            title={layer.title}
          >
            <span />
          </div>
        ))}
      </div>
    </section>
  );
}
