import React, { Component } from 'react'

class Form extends Component {

    constructor(props) {
        super(props)

        this.state = {
            username: '',
            comments: '',
            topic: 'A'
        }
    }

    handleUsernameChange = (event) => {
        this.setState({
            username: event.target.value
        })
    }

    handleCommentsChange = (event) => {
        this.setState({
            comments: event.target.value
        })
    }

    handleTopicChange = (event) => {
        this.setState({
            topic: event.target.value
        })
    }

    handleSubmit = event => {
        event.preventDefault();
        alert(`${this.state.username} ${this.state.comments} ${this.state.topic}`)
    }

    render() {
        return (
            <form onSubmit={this.handleSubmit}>
                <div>
                    <label htmlFor="">Username</label>
                    <input
                        type="text"
                        value={this.state.username}
                        onChange={this.handleUsernameChange} />
                </div>
                <div>
                    <label htmlFor="">Comment</label>
                    <textarea 
                    value={this.state.comments}
                    onChange={this.handleCommentsChange}
                    cols="30" 
                    rows="10"></textarea>
                </div>
                <div>
                    <label htmlFor="">Topic</label>
                    <select 
                        value={this.state.topic}
                        onChange={this.handleTopicChange}>
                        <option value="A">A</option>
                        <option value="B">B</option>
                        <option value="C">C</option>
                    </select>
                </div>

                <button>Submit</button>
            </form>
        )
    }
}

export default Form
