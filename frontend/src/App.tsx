import React, {useState, useEffect} from 'react';
import './App.css';

const App = () => {

  const [message, setMessage] = useState<string>();

  const fetchData = async (route: string) => {
    const response = await (
      await fetch(route,{
        headers:{
            "accepts":"application/json"
        }
    })
    ).json();
    setMessage(response.message);

  }

  useEffect(() => {
    fetchData("http://localhost:8000/");
  });

  if (!message)
    return null;

  return (
    <p>{message}</p>
  );
}

export default App;
