import React, {useState, useEffect} from 'react';
import './App.css';

const App = () => {

  const [message, setMessage] = useState<string>();

  const fetchData = async (route: string) => {
    const response = await (
      await fetch(route)
    ).json();
    setMessage(response.message);

  }

  useEffect(() => {
    console.log('Fetching data');
    fetchData('/')
    console.log("data", message);
  })

  if (message)
    return null;

  return (
    <p>{message}</p>
  );
}

export default App;
