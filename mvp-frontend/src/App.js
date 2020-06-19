import React from 'react';
import './App.css';
import AnalysisComponent from './AnalysisComponent';
import ResultsComponent from './ResultsComponent';
import Button from 'react-bootstrap/Button';

class App extends React.Component {

  constructor() {
    super()
    this.analyseVideo = this.analyseVideo.bind(this)
    this.changeURLInputHandler = this.changeURLInputHandler.bind(this)
    this.state = {
      analysingVideo: false,
      displayAnalysisResults: false,
      analysisResult: '',
      currentProbability: '0.0',
      urlInput: '',
      analysisProgress: 0
    }
  }

  analyseVideo (e) {
    e.preventDefault()
    console.log("running analysis")
    console.log(this.state.urlInput)

    this.setState({
      analysingVideo: true,
      displayAnalysisResults: false
    })

    var xhr = new XMLHttpRequest()
    var last_response_length = 0
    var length_difference = 0
    xhr.addEventListener('load', () => {
      console.log("yo")
      console.log(xhr.responseText)
    })
    xhr.addEventListener('progress', () => {
      if (xhr.responseText.substr(xhr.responseText.length - 4) == "Fake") {
        this.setState({
          analysingVideo: false,
          displayAnalysisResults: true,
          analysisResult: 'Fake',
          analysisProgress: 0,
          currentProbability: '0.0'
        })
        return
      } 
      if (xhr.responseText.substr(xhr.responseText.length - 4) == "Real") {
        this.setState({
          analysingVideo: false,
          displayAnalysisResults: true,
          analysisResult: 'Real',
          analysisProgress: 0,
          currentProbability: '0.0'
        })
        return
      } 
      length_difference = xhr.responseText.length - last_response_length
      this.setState((prevState, props) => ({
        analysingVideo: true,
        currentProbability: xhr.responseText.substr(xhr.responseText.length - length_difference),
        analysisProgress: prevState.analysisProgress + 1
      }))
      last_response_length = xhr.responseText.length;
    })
    xhr.open('POST', 'http://127.0.0.1:5000/detect', true)
    xhr.setRequestHeader('content-type', 'application/json')
    xhr.overrideMimeType('text\/event-stream; charset=x-user-defined')
    xhr.send(JSON.stringify({ url: this.state.urlInput}))
  }

  changeURLInputHandler(e) {
    this.setState({ urlInput: e.target.value })
  } 

  render() {
    return (
    <div className="App">
      <div class="container" align="center">
        <h1>Deepfake Detector</h1>
        <br></br>
        <form onSubmit={this.analyseVideo}>
            <input type="text" id="url" class="searchbar" placeholder="Enter URL of youtube video" onChange={this.changeURLInputHandler}></input>
            <br></br>
            <br></br>
            <br></br>
            <Button variant="primary" type="submit">Analyse Video</Button>
        </form>
        <ResultsComponent 
          displayAnalysisResults={this.state.displayAnalysisResults} 
          analysisResult={this.state.analysisResult}/>
        <AnalysisComponent 
          analysisProgress={this.state.analysisProgress} 
          currentProbability={this.state.currentProbability} 
          analysingVideo={this.state.analysingVideo}/>
      </div>
    </div>
  );}
}

export default App;
