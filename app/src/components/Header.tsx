import { useSetAtom } from "jotai";
import { MdMenu } from "react-icons/md";

import { isMenuOpenAtom } from "../stores/Navigation";

import { useLocation } from "react-router-dom";
import { getAboutPath } from "../views/App/urls";
import "./Header.scss";

function Header() {
  const location = useLocation();
  const setIsMenuOpen = useSetAtom(isMenuOpenAtom);

  return (
    <header>
      <h1>
        <MdMenu className="icon" onClick={() => setIsMenuOpen((isOpen) => !isOpen)} />
        <span>GPT Circuits</span>
      </h1>
      {location.pathname !== getAboutPath() && <a href={`#${getAboutPath()}`}>How This Works</a>}
    </header>
  );
}

export { Header };
