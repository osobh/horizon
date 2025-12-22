import React from 'react'
import { TopologyEditor } from './components/TopologyEditor'

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>StratoSwarm Visual Editor</h1>
        <p>GPU-native distributed system topology designer</p>
      </header>
      <main className="app-main">
        <TopologyEditor />
      </main>
    </div>
  )
}

export default App