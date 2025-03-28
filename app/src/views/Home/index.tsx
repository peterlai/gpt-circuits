import { FeatureExtraction } from "../../components/Home/FeatureExtraction";
import { Intro } from "../../components/Home/Intro";
import "./style.scss";

function Home() {
  return (
    <div id="Home">
      <main>
        <Intro />
        <FeatureExtraction />
      </main>
    </div>
  );
}

export { Home };
