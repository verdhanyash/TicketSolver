// Quality/Eval panel (FR-16.3) — placeholder until the eval harness lands (FR-12/13/17).
// Will show resolution/escalation correctness plus ML model metrics + feature importance (FR-18).
export function QualityEvalPanel() {
  return (
    <section className="panel">
      <h2>Accuracy & Quality</h2>
      <p>No data yet — this will show how often the system resolves tickets
        correctly and when it correctly asks a human for help.</p>
    </section>
  );
}
