import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { loader } from '@monaco-editor/react';
import App from './App';
import './index.css';
import { registerSwarmLanguage, swarmThemeRules } from './languages/swarm';

// Register custom languages and themes when Monaco loads
loader.init().then((monaco) => {
  // Register .swarm DSL language
  registerSwarmLanguage();

  // Extend vs-dark theme with .swarm token colors
  monaco.editor.defineTheme('horizon-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: swarmThemeRules,
    colors: {
      'editor.background': '#0f172a', // slate-900
    },
  });
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
