import { getAboutPath } from "../App/urls";
import "./style.scss";

function NotFound() {
  return (
    <main id="NotFound">
      <h1>Not Found</h1>
      <p>We can't find what you're looking for.</p>
      <p>
        <a href={`#${getAboutPath()}`}>Return Home »</a>
      </p>
    </main>
  );
}

export { NotFound };
