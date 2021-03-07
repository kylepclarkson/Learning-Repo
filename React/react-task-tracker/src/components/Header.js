


const Header = (props) => {
    return (
        <div className='container'>
            <h1>Task Tracker - {props.title} </h1>
        </div>
    )
}

Header.defailtProps = {
    title: 'Greeting'
}

export default Header
