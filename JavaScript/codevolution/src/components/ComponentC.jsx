import React, { Component } from 'react'
import { UserConsumer } from './UserContext'

class ComponentC extends Component {
    render() {
        return (
            // access value defined in context.
            <UserConsumer>
                {
                    (username) => {
                        return <div>Hello {username}</div>
                    }
                }
            </UserConsumer>
        )
    }
}

export default ComponentC
