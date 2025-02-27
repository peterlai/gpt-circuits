import "./style.scss";

function Home() {
  return (
    <div id="Home">
      <main>
        <section className="intro">
          <h1>
            <img src={`${process.env.PUBLIC_URL}/logo.png`} alt="logo" />
            <span>GPT-2 Circuits</span>
          </h1>
        </section>
      </main>
    </div>
  );
}

export { Home };
