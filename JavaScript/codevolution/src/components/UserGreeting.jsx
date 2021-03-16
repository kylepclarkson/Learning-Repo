import React, { Component } from 'react'

class UseGreeting extends Component {

    constructor(props) {
        super(props)

        this.state = {
            isLoggedIn: false
        }
    }

    render() {
        
        let message
        message = (this.state.isLoggedIn) ? 'Hello user' : 'Hello guest'

        return (
            <div>
                {message}
            </div>
        )

        // cannot mix conditional and jsx
        // if (this.state.isLoggedIn) {
        //     return (
        //         <div>Welcome user!</div>
        //     )
        // } else {
        //     <div>Welcome Guest!</div>
        // }
    }
}

export default UseGreeting
