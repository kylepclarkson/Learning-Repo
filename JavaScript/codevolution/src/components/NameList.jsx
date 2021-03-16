import React from 'react'

import Person from './Person'

function NameList() {
    // const names = ['Alex', 'Bob', 'Charlie']

    // return (
    //     <div>
    //         {
    //             names.map(name =><h2>{name}</h2>)
    //         }
    //     </div>
    // )

    const persons = [
        {
            id: 0,
            name: 'Alex',
            age: 20
        },
        {
            id:1, 
            name: 'Bob',
            age: 21
        },
        {
            id:2,
            name: 'Charlie',
            age: 25
        }
    ]

    const personList = persons.map(person => <Person key={person.id} person={person} />)
    return <div>{personList}</div>
}

export default NameList
