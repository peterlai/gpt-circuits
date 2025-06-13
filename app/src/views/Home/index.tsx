import { Acknowledgements } from "../../components/Home/Acknowledgements";
import { Guide } from "../../components/Home/Guide";
import { Intro } from "../../components/Home/Intro";
import { VerdantForest } from "../../components/Home/VerdantForest";
import "./style.scss";

function Home() {
  return (
    <div id="Home">
      <main>
        <Intro />
        <Guide />
        <VerdantForest />
        <Acknowledgements />
      </main>
    </div>
  );
}

export { Home };
