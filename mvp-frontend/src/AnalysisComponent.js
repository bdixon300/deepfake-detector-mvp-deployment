import React from 'react';
import './App.css';
import ProgressBar from 'react-bootstrap/ProgressBar';

class AnalysisComponent extends React.Component {
  render() {
    return (
    <div className="App">
     {this.props.analysingVideo ? ((
        <div>
            <br></br>
            <p>Processing Video ... Current Probability of being real: {this.props.currentProbability}%</p>
            <ProgressBar animated now={this.props.analysisProgress} label={`${(Math.round((this.props.analysisProgress / 35) * 100))}%`} max={35}/>
            <br></br>
        </div>
        )) 
        : (null)}
    </div>
  );}
}

export default AnalysisComponent;
