import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useEffect } from "react";
import { BiArrowFromLeft } from "react-icons/bi";

import { isMenuOpenAtom, isMobile, isSidebarOpenAtom } from "../../../stores/Navigation";
import { predictionDataAtom } from "../../../stores/Prediction";
import { selectionStateAtom } from "../../../stores/Selection";
import { FeatureSidebar } from "./features";
import { PredictionSidebar } from "./prediction";

import "charts.css";
import { BlockSidebar } from "./blocks";
import "./style.scss";

function Sidebar() {
  const { focusedBlock, selectedBlock, focusedFeature, selectedFeature } =
    useAtomValue(selectionStateAtom);
  const [{ isPending, isError }] = useAtom(predictionDataAtom);
  const setIsSidebarOpen = useSetAtom(isSidebarOpenAtom);
  const setIsMenuOpen = useSetAtom(isMenuOpenAtom);

  // Open the sidebar when a feature is selected
  useEffect(() => {
    if (selectedBlock || selectedFeature) {
      setIsSidebarOpen(true);

      // Also close the menu on mobile
      if (isMobile()) {
        setIsMenuOpen(false);
      }
    }
  }, [selectedBlock, selectedFeature, setIsSidebarOpen, setIsMenuOpen]);

  if (isPending || isError) {
    return <></>;
  }

  return (
    <div id="Sidebar">
      {focusedBlock || focusedFeature ? null : (
        <BiArrowFromLeft className="close-sidebar" onClick={() => setIsSidebarOpen(false)} />
      )}
      {focusedBlock ? <BlockSidebar block={focusedBlock} /> : null}
      {focusedFeature ? <FeatureSidebar feature={focusedFeature} /> : null}
      {!focusedBlock && !focusedFeature ? <PredictionSidebar /> : null}
    </div>
  );
}

export { Sidebar };
