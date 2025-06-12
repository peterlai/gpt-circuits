import "./Acknowledgements.scss";

function Acknowledgements() {
  return (
    <section id="Acknowledgements">
      <h2>Acknowledgements</h2>
      <p>
        This project was completed with support from the{" "}
        <a href="https://www.cambridgeaisafety.org" target="_blank" rel="noopener noreferrer">
          Cambridge AI Safety Hub
        </a>{" "}
        and with mentorship from{" "}
        <a href="https://heimersheim.eu" target="_blank" rel="noopener noreferrer">
          Stefan Heimersheim
        </a>
        . It builds upon{" "}
        <a href="https://peterlai.github.io/gpt-mri/" target="_blank" rel="noopener noreferrer">
          prior work
        </a>{" "}
        extracting circuits from 4- & 6-layer character-based GPT-2 models.
      </p>
    </section>
  );
}

export { Acknowledgements };
