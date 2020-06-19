import React from 'react';
import './App.css';

class ResultsComponent extends React.Component {
  render() {
    return (
    <div className="App">
    <br></br>
    <br></br>
     {this.props.displayAnalysisResults ? ((<p>This video is: {this.props.analysisResult}</p>) ) : (null)}
    </div>
  );}
}

export default ResultsComponent;
