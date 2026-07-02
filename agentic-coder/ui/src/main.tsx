import ReactDOM from 'react-dom/client'
import App from './App'
import 'highlight.js/styles/vs2015.css'
import 'xterm/css/xterm.css'
import './index.css'

// No StrictMode: it double-mounts effects in dev, which would open a second
// EventSource and re-init xterm instances. Production behaviour is unaffected.
ReactDOM.createRoot(document.getElementById('root')!).render(<App />)
