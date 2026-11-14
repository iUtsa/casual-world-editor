import type { Asset } from "../types";

const PROC_ICON: Record<string, string> = { domino: "▮", marble: "●", ramp: "◺", steel_ball: "●" };

export default function Library({ assets, chosen, onChoose, disabled }: {
  assets: Asset[]; chosen: string | null; onChoose: (id: string | null) => void; disabled: boolean;
}) {
  return (
    <section className="panel">
      <h3>Object library <span className="muted small">click, then click the table</span></h3>
      <div className="lib-grid">
        {assets.map((a) => (
          <button key={a.id} disabled={disabled} className={`lib-item ${chosen === a.id ? "on" : ""}`}
            title={`${a.name} · ${a.dims.map((d) => (d * 100).toFixed(1)).join(" × ")} cm · ${Math.round(a.mass * 1000)} g` +
              (a.license ? ` · ${a.license}${a.authors?.length ? " by " + a.authors.join(", ") : ""}` : "")}
            onClick={() => onChoose(chosen === a.id ? null : a.id)}>
            {a.thumb ? <img src={a.thumb} alt="" /> : <span className="proc">{PROC_ICON[a.id] ?? "■"}</span>}
            <span>{a.name}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
