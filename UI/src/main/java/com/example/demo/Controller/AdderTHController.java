package com.example.demo.Controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

@Controller
public class AdderTHController {
    @RequestMapping("/")
    public String formPage() {
        return "view/main-1/index";
    }

    @PostMapping("/table")
    public String formSubmittable(@RequestParam("search") String search, Model model) {
        model.addAttribute("search", search);
        return "view/table/index";
    }
    @PostMapping("/map")
    public String formSubmitlist(@RequestParam("search") String search, Model model) {
        model.addAttribute("search", search);
        return "view/map/index";
    }
    @PostMapping("/upload")
    public ResponseEntity<String> uploadFile(@RequestParam("file") MultipartFile file) {
        String UPLOAD_DIR = "C:\\capstone\\userImg";

        if (file.isEmpty()) {
            return new ResponseEntity<>("No file selected", HttpStatus.BAD_REQUEST);
        }

        // 이미지 파일 MIME 타입 체크
        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            return new ResponseEntity<>("Only image files are allowed", HttpStatus.BAD_REQUEST);
        }

        // 업로드 디렉토리 생성
        File directory = new File(UPLOAD_DIR);
        if (!directory.exists()) {
            directory.mkdirs();
        }

        try {
            // 파일 저장 경로 설정
            String filePath = UPLOAD_DIR + "\\" + file.getOriginalFilename();
            file.transferTo(new File(filePath));
            return new ResponseEntity<>(file.getOriginalFilename(), HttpStatus.OK);
        } catch (IOException e) {
            e.printStackTrace();
            return new ResponseEntity<>("File upload failed", HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
}
