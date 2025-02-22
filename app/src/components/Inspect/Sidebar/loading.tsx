import { CgSpinner } from "react-icons/cg";
import { FaX } from "react-icons/fa6";

function LoadingMessage() {
  return (
    <section>
      <h3 className="loading">
        <CgSpinner className="icon spin" />
        Loading
      </h3>
    </section>
  );
}

function ErrorMessage() {
  return (
    <section>
      <h3 className="error">
        <FaX className="icon" />
        Error
      </h3>
    </section>
  );
}

export { ErrorMessage, LoadingMessage };
