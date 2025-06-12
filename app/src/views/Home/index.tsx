import { Acknowledgements } from "../../components/Home/Acknowledgements";
import { Guide } from "../../components/Home/Guide";
import { Intro } from "../../components/Home/Intro";
import "./style.scss";

function Home() {
  return (
    <div id="Home">
      <main>
        <Intro />
        <Guide />
        <Acknowledgements />
      </main>
    </div>
  );
}

export { Home };
