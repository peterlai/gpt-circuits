import { useSetAtom } from "jotai";
import { MdClose } from "react-icons/md";
import { isMobile, isSidebarOpenAtom } from "../../../stores/Navigation";
import { toggleSelectionAtom } from "../../../stores/Selection";

// This component is used to close the sidebar and clear the selection.
function CloseButton() {
  const setIsSidebarOpen = useSetAtom(isSidebarOpenAtom);
  const toggleSelection = useSetAtom(toggleSelectionAtom);

  const closeHandler = () => {
    if (isMobile()) {
      // Wait for animation to finish before clearing the selection.
      setIsSidebarOpen(false);
      setTimeout(() => {
        toggleSelection(null);
      }, 300);
    } else {
      toggleSelection(null);
    }
  };

  return <MdClose className="close" onClick={closeHandler} />;
}

export { CloseButton };
