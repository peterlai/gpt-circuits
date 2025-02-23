import { useAtomValue, useSetAtom } from "jotai";
import { useEffect } from "react";
import { useParams } from "react-router-dom";

import { AblationMap } from "../../components/Inspect/AblationMap";
import { Sidebar } from "../../components/Inspect/Sidebar";
import { Navbar } from "../../components/Navbar";
import { modelIdAtom, sampleIdAtom } from "../../stores/Graph";
import { isSidebarOpenAtom } from "../../stores/Navigation";
import { toggleSelectionAtom } from "../../stores/Selection";

import "./style.scss";

function Inspect() {
  const {
    modelId: modelIdFromUrl,
    sampleId: sampleIdFromUrl,
    selectionKey: selectionKeyFromUrl,
  } = useParams();
  const setSampleId = useSetAtom(sampleIdAtom);
  const setModelId = useSetAtom(modelIdAtom);
  const toggleSelection = useSetAtom(toggleSelectionAtom);
  const isSidebarOpen = useAtomValue(isSidebarOpenAtom);

  // Reset selection state if sample ID changes.
  useEffect(() => {
    toggleSelection(null);
  }, [sampleIdFromUrl, toggleSelection]);

  // Set sample ID and feature from URL params.
  useEffect(() => {
    setModelId(modelIdFromUrl ?? "");
    setSampleId(sampleIdFromUrl ?? "");
  }, [modelIdFromUrl, sampleIdFromUrl, setModelId, setSampleId]);

  // Cleanup function to be called upon unmount.
  useEffect(() => {
    return () => {
      setModelId("");
      setSampleId("");
    };
  }, [setModelId, setSampleId]);

  // Update the body class to show/hide the sidebar.
  useEffect(() => {
    document.body.classList.toggle("sidebar-open", isSidebarOpen);
  }, [isSidebarOpen]);

  return (
    <>
      <div id="Inspect">
        <nav>
          <Navbar />
        </nav>
        <main>
          <AblationMap selectionKeyFromUrl={selectionKeyFromUrl} />
        </main>
        <aside>
          <Sidebar />
        </aside>
      </div>
    </>
  );
}

export default Inspect;
