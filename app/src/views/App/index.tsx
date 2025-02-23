import { useAtomValue } from "jotai";
import { useEffect } from "react";
import { HashRouter, Route, Routes } from "react-router-dom";

import { Header } from "../../components/Header";
import { Menu } from "../../components/Menu";
import { isMenuOpenAtom } from "../../stores/Navigation";
import Embed from "../Embed";
import Inspect from "../Inspect";
import { NotFound } from "../NotFound";

import { Home } from "../Home";
import "./style.scss";

function App() {
  const isMenuOpen = useAtomValue(isMenuOpenAtom);

  // Update the body class to show/hide the menu.
  useEffect(() => {
    document.body.classList.toggle("menu-open", isMenuOpen);
  }, [isMenuOpen]);

  // Delay adding the "loaded" class to avoid animations on page load.
  useEffect(() => {
    setTimeout(() => {
      document.body.classList.toggle("loaded", true);
    }, 1);
  }, []);

  return (
    <HashRouter>
      <Header />
      <article>
        <Routes>
          <Route path="/" element=<Home /> />
          <Route path="/samples/:modelId/:sampleId/:selectionKey?" element=<Inspect /> />
          <Route path="/embedded/:modelId/:sampleId/:selectionKey?" element=<Embed /> />
          <Route path="*" element=<NotFound /> />
        </Routes>
      </article>
      <Menu />
    </HashRouter>
  );
}

export default App;
