import React, { useState } from 'react';
import axios from 'axios';

const Form = ({ onResults }) => {
    const [data, setData] = useState('');
    const [fcmType, setFcmType] = useState('Unsupervised FCM');
    const [clusters, setClusters] = useState(2);
    const [m, setM] = useState(2.0);
    const [eps, setEps] = useState(0.01);
    const [lmax, setLmax] = useState(50);
    const [alpha, setAlpha] = useState(0.5);
    const [beta, setBeta] = useState(1.0);
    const [uSupervised, setUSupervised] = useState('');
    const [file, setFile] = useState(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = (event) => {
            setData(event.target.result);
        };
        reader.readAsText(file);
        setFile(file);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const payload = {
            data,
            fcm_type: fcmType,
            clusters,
            m,
            eps,
            lmax,
            alpha: fcmType !== "Unsupervised FCM" ? alpha : undefined,
            beta: fcmType === "Entropy Regularized FCM" ? beta : undefined,
            u_supervised: fcmType !== "Unsupervised FCM" && uSupervised ? JSON.parse(uSupervised) : undefined,
        };

        try {
            const response = await axios.post('http://127.0.0.1:5000/run_fcm', payload);
            onResults(response.data);
        } catch (error) {
            console.error('Error running FCM algorithm:', error);
        }
    };

    return (
        <div className="container">
            <h2>FCM Algorithm Demo</h2>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label>Data (CSV format):</label>
                    <input type="file" onChange={handleFileChange} accept=".csv" required />
                </div>
            <div>
                <label>FCM Type:</label>
                <select value={fcmType} onChange={(e) => setFcmType(e.target.value)}>
                    <option value="Unsupervised FCM">Unsupervised FCM</option>
                    <option value="Semi-Supervised FCM">Semi-Supervised FCM</option>
                    <option value="Entropy Regularized FCM">Entropy Regularized FCM</option>
                </select>
            </div>
            <div>
                <label>Number of Clusters:</label>
                <input type="number" value={clusters} onChange={(e) => setClusters(e.target.value)} min="2" max="100" required />
            </div>
            <div>
                <label>Fuzziness Coefficient (m):</label>
                <input type="number" value={m} onChange={(e) => setM(e.target.value)} min="1.1" max="5.0" step="0.1" required />
            </div>
            <div>
                <label>Epsilon:</label>
                <input type="number" value={eps} onChange={(e) => setEps(e.target.value)} min="0.001" max="0.1" step="0.001" required />
            </div>
            <div>
                <label>Maximum Number of Iterations:</label>
                <input type="number" value={lmax} onChange={(e) => setLmax(e.target.value)} min="10" max="100" required />
            </div>
            {fcmType !== "Unsupervised FCM" && (
                <>
                    <div>
                        <label>Alpha (Weight for supervised information):</label>
                        <input type="number" value={alpha} onChange={(e) => setAlpha(e.target.value)} min="0.0" max="1.0" step="0.1" required />
                    </div>
                    {fcmType === "Entropy Regularized FCM" && (
                        <div>
                            <label>Beta (Weight for entropy regularization):</label>
                            <input type="number" value={beta} onChange={(e) => setBeta(e.target.value)} min="0.0" max="5.0" step="0.1" required />
                        </div>
                    )}
                    <div>
                        <label>Supervised Membership Data (JSON format):</label>
                        <textarea value={uSupervised} onChange={(e) => setUSupervised(e.target.value)} />
                    </div>
                </>
            )}
            <button className="btn-primary" type="submit">Run Algorithm</button>
        </form>
        </div>
    );
};

export default Form;
