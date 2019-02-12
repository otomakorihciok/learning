package lib

import (
	"github.com/jinzhu/gorm"
)

// User is user model.
type User struct {
	gorm.Model
	Name   string
	Age    int
	Gender string
}
