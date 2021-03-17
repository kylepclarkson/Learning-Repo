import React, { Component } from 'react'

// Errorboundaries only occur in production. 
class ErrorBoundary extends Component {

    constructor(props) {
        super(props)
    
        this.state = {
             hasError: false
        }
    }
    

    static getDerivedStateFromError(error) {
        return {
            hasError: true
        }
    }

    render() {
        if (this.state.hasError) {
            return <h1>Something went wrong</h1>
        }
        return this.props.children
    }
}

export default ErrorBoundary
