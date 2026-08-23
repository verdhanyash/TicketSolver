import { Canvas } from "@react-three/fiber";
import type { Trace } from "../../types";

// Interactive agent-trace graph (FR-11, FR-19): nodes = agents/tool calls,
// edges = decision flow, animated replay. Scaffold renders an empty canvas;
// the graph layout + replay animation arrive in a later session.
export function TraceViewer({ trace }: { trace?: Trace }) {
  return (
    <div className="trace-viewer" style={{ width: "100%", height: 420 }}>
      <Canvas camera={{ position: [0, 0, 12] }}>
        <ambientLight />
        {/* Placeholder scene — TraceNode meshes + animated edges land later. */}
        {trace ? null : null}
      </Canvas>
    </div>
  );
}
