import { useRef, useState, useEffect } from "react";
import { motion } from "motion/react";

export default function App() {
  // Create a reference to the worker object.
  const worker = useRef(null);

  const [inputText, setInputText] = useState("Life is like a box of chocolates. You never know what you're gonna get.");
  const [selectedSpeaker, setSelectedSpeaker] = useState("af_heart");

  const [voices, setVoices] = useState([]);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState("Loading...");

  const [results, setResults] = useState([]);
  const [chunks, setChunks] = useState([]);

  // We use the `useEffect` hook to setup the worker as soon as the `App` component is mounted.
  useEffect(() => {
    // Create the worker if it does not yet exist.
    worker.current ??= new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case "device":
          setLoadingMessage(`Loading model (device="${e.data.device}")`);
          break;
        case "ready":
          setStatus("ready");
          setVoices(e.data.voices);
          break;
        case "error":
          setError(e.data.data);
          setStatus("ready"); // Reset status
          setChunks([]); // Clear any partial chunks
          break;
        case "chunk":
          // When the first chunk arrives, create a placeholder UI
          if (e.data.first) {
            setResults((prev) => [{ text: e.data.text, src: null }, ...prev]);
          }
          // Collect the blobs
          setChunks((prev) => [...prev, e.data.audio]);
          break;
        case "complete":
          // Generation complete: merge blobs and play
          const fullBlob = new Blob(chunks, { type: "audio/wav" });
          const audioUrl = URL.createObjectURL(fullBlob);

          // Update the result with the final URL
          setResults((prev) => {
            const newResults = [...prev];
            newResults[0].src = audioUrl;
            return newResults;
          });
          setChunks([]); // Clear chunks for the next run
          setStatus("ready");
          break;
      }
    };

    const onErrorReceived = (e) => {
      console.error("Worker error:", e);
      setError(e.message);
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener("message", onMessageReceived);
    worker.current.addEventListener("error", onErrorReceived);

    // Define a cleanup function for when the component is unmounted.
    return () => {
      worker.current.removeEventListener("message", onMessageReceived);
      worker.current.removeEventListener("error", onErrorReceived);
    };
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    setStatus("running");

    worker.current.postMessage({
      type: "generate",
      text: inputText.trim(),
      voice: selectedSpeaker,
    });
  };

  return (
    <div className="relative w-full min-h-screen bg-gradient-to-br from-gray-900 to-gray-700 flex flex-col items-center justify-center p-4 relative overflow-hidden font-sans">
      <motion.div initial={{ opacity: 1 }} animate={{ opacity: status === null ? 1 : 0 }} transition={{ duration: 0.5 }} className="absolute w-screen h-screen justify-center flex flex-col items-center z-10 bg-gray-800/95 backdrop-blur-md" style={{ pointerEvents: status === null ? "auto" : "none" }}>
        <div className="w-[250px] h-[250px] border-4 border-white shadow-[0_0_0_5px_#4973ff] rounded-full overflow-hidden">
          <div className="loading-wave"></div>
        </div>
        <p className={`text-3xl my-5 text-center ${error ? "text-red-500" : "text-white"}`}>{error ?? loadingMessage}</p>
      </motion.div>

      <div className="max-w-3xl w-full space-y-8 relative z-[2]">
        <div className="text-center">
          <h1 className="text-5xl font-extrabold text-gray-100 mb-2 drop-shadow-lg font-heading">Kokoro Text-to-Speech</h1>
          <p className="text-2xl text-gray-300 font-semibold font-subheading">
            Powered by&nbsp;
            <a href="https://github.com/hexgrad/kokoro" target="_blank" rel="noreferrer" className="underline">
              Kokoro
            </a>
            &nbsp;and&nbsp;
            <a href="https://huggingface.co/docs/transformers.js" target="_blank" rel="noreferrer" className="underline">
              <img width="40" src="hf-logo.svg" className="inline translate-y-[-2px] me-1"></img>Transformers.js
            </a>
          </p>
        </div>
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <textarea placeholder="Enter text..." value={inputText} onChange={(e) => setInputText(e.target.value)} className="w-full min-h-[100px] max-h-[300px] bg-gray-700/50 backdrop-blur-sm border-2 border-gray-600 rounded-xl resize-y text-gray-100 placeholder-gray-400 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent" rows={Math.min(8, inputText.split("\n").length)} />
            <div className="flex flex-col items-center space-y-4">
              <select value={selectedSpeaker} onChange={(e) => setSelectedSpeaker(e.target.value)} className="w-full bg-gray-700/50 backdrop-blur-sm border-2 border-gray-600 rounded-xl text-gray-100 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                {Object.entries(voices).map(([id, voice]) => (
                  <option key={id} value={id}>
                    {voice.name} ({voice.language === "en-us" ? "American" : "British"} {voice.gender})
                  </option>
                ))}
              </select>
              <button type="submit" className="inline-flex justify-center items-center px-6 py-2 text-lg font-semibold bg-gradient-to-t from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 transition-colors duration-300 rounded-xl text-white disabled:opacity-50" disabled={status === "running" || inputText.trim() === ""}>
                {status === "running" ? "Generating..." : "Generate"}
              </button>
            </div>
          </form>
        </div>

        {results.length > 0 && (
          <motion.div initial={{ y: 50, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.5 }} className="max-h-[250px] overflow-y-auto px-2 mt-4 space-y-6 relative z-[2]">
            {results.map((result, i) => (
              <div key={i}>
                <div className="text-white bg-gray-800/70 backdrop-blur-sm border border-gray-700 rounded-lg p-4 z-10">
                  <span className="absolute right-5 font-bold">#{results.length - i}</span>
                  <p className="mb-3 max-w-[95%]">{result.text}</p>
                  {result.src === null ? (
                    <div className="w-full h-8 bg-gray-600 rounded-lg animate-pulse"></div>
                  ) : (
                    <audio controls src={result.src} autoPlay className="w-full">
                      Your browser does not support the audio element.
                    </audio>
                  )}
                </div>
              </div>
            ))}
          </motion.div>
        )}
      </div>

      <div className="bg-[#015871] pointer-events-none absolute left-0 w-full h-[5%] bottom-[-50px]">
        <div className="wave"></div>
        <div className="wave"></div>
      </div>
    </div>
  );
}
