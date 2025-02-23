import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useEffect } from "react";
import { BiArrowFromLeft } from "react-icons/bi";

import { isMenuOpenAtom, isMobile, isSidebarOpenAtom } from "../../../stores/Navigation";
import { predictionDataAtom } from "../../../stores/Prediction";
import { selectionStateAtom } from "../../../stores/Selection";
import { FeatureSidebar } from "./features";
import { PredictionSidebar } from "./prediction";

import "charts.css";
import "./style.scss";

function Sidebar() {
  const { focusedFeature, selectedFeature } = useAtomValue(selectionStateAtom);
  const [{ isPending, isError }] = useAtom(predictionDataAtom);
  const setIsSidebarOpen = useSetAtom(isSidebarOpenAtom);
  const setIsMenuOpen = useSetAtom(isMenuOpenAtom);

  // Open the sidebar when a feature is selected
  useEffect(() => {
    if (selectedFeature) {
      setIsSidebarOpen(true);

      // Also close the menu on mobile
      if (isMobile()) {
        setIsMenuOpen(false);
      }
    }
  }, [selectedFeature, setIsSidebarOpen, setIsMenuOpen]);

  if (isPending || isError) {
    return <></>;
  }

  return (
    <div id="Sidebar">
      {!focusedFeature && (
        <BiArrowFromLeft className="close-sidebar" onClick={() => setIsSidebarOpen(false)} />
      )}
      {focusedFeature ? <FeatureSidebar feature={focusedFeature} /> : <PredictionSidebar />}
    </div>
  );
}

export { Sidebar };
