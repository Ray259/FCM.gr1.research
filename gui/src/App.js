// src/App.js
import React, { useState } from 'react';
import Form from './Form';
import './App.css'; // Import CSS file

const App = () => {
    const [results, setResults] = useState(null);

    const handleResults = (data) => {
        setResults(data);
    };

    return (
        <div className="container">
            <h1>FCM Algorithm Demo</h1>
            <Form onResults={handleResults} />
            {results && (
                <div className="results">
                    <h2>Results</h2>
                    <pre>{JSON.stringify(results, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default App;
