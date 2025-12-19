import { useEffect, useMemo, useState } from "react";
import "./App.css";

type Variant = {
  label: string;
  path: string;
  data_url: string;
};

type PreviewResponse = {
  image: string;
  variants: Variant[];
};

type SelectionItem = {
  label: string;
  path: string;
};

type LutIndex = {
  labels: string[];
  count: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [imagePath, setImagePath] = useState("");
  const [limit, setLimit] = useState<string>("");
  const [variants, setVariants] = useState<Variant[]>([]);
  const [imageName, setImageName] = useState<string | null>(null);
  const [selected, setSelected] = useState<Map<string, SelectionItem>>(
    () => new Map()
  );
  const [modalVariant, setModalVariant] = useState<Variant | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [lutsIndex, setLutsIndex] = useState<LutIndex | null>(null);

  const selectedItems = useMemo(
    () => Array.from(selected.values()),
    [selected]
  );

  useEffect(() => {
    const fetchLuts = async () => {
      try {
        const res = await fetch(`${API_BASE}/luts`);
        if (!res.ok) return;
        const data: LutIndex = await res.json();
        setLutsIndex(data);
      } catch (err) {
        console.warn("Unable to fetch LUT list", err);
      }
    };
    fetchLuts();
  }, []);

  const handlePreview = async () => {
    if (!file && !imagePath.trim()) {
      setError("Upload an image or provide a server-side image path.");
      return;
    }

    const form = new FormData();
    if (file) form.append("image_file", file);
    if (imagePath.trim()) form.append("image_path", imagePath.trim());
    if (limit.trim()) form.append("limit", limit.trim());

    setLoading(true);
    setError(null);
    setStatus("Generating previews...");
    try {
      const res = await fetch(`${API_BASE}/preview`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || "Preview failed");
      }
      const data: PreviewResponse = await res.json();
      setVariants(data.variants || []);
      setImageName(data.image || file?.name || imagePath);
      setSelected(new Map());
      setStatus(
        `Generated ${data.variants?.length ?? 0} previews for ${data.image ?? "image"}.`
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Preview failed.";
      setError(message);
      setStatus(null);
    } finally {
      setLoading(false);
    }
  };

  const toggleLabel = (item: SelectionItem) => {
    setSelected((prev) => {
      const next = new Map(prev);
      if (next.has(item.label)) {
        next.delete(item.label);
      } else {
        next.set(item.label, item);
      }
      return next;
    });
  };

  const clearSelection = () => setSelected(new Map());

  const handleExport = async () => {
    if (!selected.size) {
      setError("Pick at least one LUT label to export.");
      return;
    }
    const payload = {
      image: imageName || file?.name || imagePath || "image",
      selection: selectedItems,
    };
    setLoading(true);
    setError(null);
    setStatus("Exporting selection...");
    try {
      const res = await fetch(`${API_BASE}/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || "Export failed");
      }
      const data = await res.json();
      setStatus(`Selection saved to ${data.written_to ?? "server"}.`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Export failed.";
      setError(message);
      setStatus(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">LUT picker</p>
          <h1>Preview filters, label favorites, export JSON.</h1>
          <p className="lede">
            Upload or point to an image on the server. We&apos;ll apply every LUT in your
            folder, show the previews, and let you export the chosen filter names.
          </p>
          {lutsIndex ? (
            <p className="meta">
              {lutsIndex.count} LUT file{lutsIndex.count === 1 ? "" : "s"} detected in
              ml_luts.
            </p>
          ) : (
            <p className="meta">LUT inventory will load when the API is reachable.</p>
          )}
        </div>
      </header>

      <section className="panel">
        <div className="fields">
          <label className="field">
            <span>Upload image</span>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                const next = e.target.files?.[0];
                setFile(next ?? null);
                if (next) setImagePath("");
              }}
            />
          </label>

          <label className="field">
            <span>Or server image path</span>
            <input
              type="text"
              placeholder="processed_images/example.png"
              value={imagePath}
              onChange={(e) => {
                setImagePath(e.target.value);
                if (e.target.value) setFile(null);
              }}
            />
          </label>

          <label className="field">
            <span>Limit (optional)</span>
            <input
              type="number"
              min={1}
              max={200}
              value={limit}
              placeholder="Leave blank to use all LUTs"
              onChange={(e) => setLimit(e.target.value)}
            />
          </label>
        </div>

        <div className="actions">
          <button className="primary" onClick={handlePreview} disabled={loading}>
            {loading ? "Working..." : "Generate previews"}
          </button>
          <button onClick={handleExport} disabled={loading || !selected.size}>
            Export selection
          </button>
          <button onClick={clearSelection} disabled={!selected.size || loading}>
            Clear selection
          </button>
        </div>

        {status && <div className="status">{status}</div>}
        {error && <div className="error">{error}</div>}
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Preview grid</p>
            <h2>
              {variants.length ? `${variants.length} variants` : "No previews yet"}
            </h2>
            {imageName && <p className="meta">Source: {imageName}</p>}
          </div>
      <div className="selection-pill">
        <span>{selected.size}</span> selected
      </div>
        </div>

        {variants.length === 0 && (
          <p className="muted">Run a preview to see your LUTs applied.</p>
        )}

        <div className="grid">
          {variants.map((variant) => {
            const active = selected.has(variant.label);
            return (
              <div
                key={variant.label}
                className={`card ${active ? "card-active" : ""}`}
                onClick={() => toggleLabel({ label: variant.label, path: variant.path })}
              >
                <div className="image-frame">
                  <img src={variant.data_url} alt={variant.label} />
                </div>
                <div className="card-footer">
                  <span className="label ellipsis" title={variant.label}>
                    {variant.label}
                  </span>
                  <div className="pill-row">
                    <span className={`pill ${active ? "pill-active" : ""}`}>
                      {active ? "Picked" : "Pick"}
                    </span>
                    <button
                      className="ghost"
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setModalVariant(variant);
                      }}
                    >
                      View
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {modalVariant && (
          <div className="modal-backdrop" onClick={() => setModalVariant(null)}>
            <div
              className="modal"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="modal-header">
                <span className="label">{modalVariant.label}</span>
                <button className="ghost" onClick={() => setModalVariant(null)}>
                  Close
                </button>
              </div>
              <div className="modal-body">
                <img src={modalVariant.data_url} alt={modalVariant.label} />
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export default App;
